import subprocess
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from packaging import version
from .. import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer
from ..utils import logging
from ..utils.import_utils import is_optimum_available
from .convert import export, validate_model_outputs
from .features import FeaturesManager
from .utils import get_preprocessor
def export_with_transformers(args):
    args.output = args.output if args.output.is_file() else args.output.joinpath('model.onnx')
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)
    model = FeaturesManager.get_model_from_feature(args.feature, args.model, framework=args.framework, cache_dir=args.cache_dir)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)
    onnx_config = model_onnx_config(model.config)
    if model_kind in ENCODER_DECODER_MODELS:
        encoder_model = model.get_encoder()
        decoder_model = model.get_decoder()
        encoder_onnx_config = onnx_config.get_encoder_config(encoder_model.config)
        decoder_onnx_config = onnx_config.get_decoder_config(encoder_model.config, decoder_model.config, feature=args.feature)
        if args.opset is None:
            args.opset = max(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)
        if args.opset < min(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset):
            raise ValueError(f'Opset {args.opset} is not sufficient to export {model_kind}. At least  {min(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)} is required.')
        preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
        onnx_inputs, onnx_outputs = export(preprocessor, encoder_model, encoder_onnx_config, args.opset, args.output.parent.joinpath('encoder_model.onnx'))
        validate_model_outputs(encoder_onnx_config, preprocessor, encoder_model, args.output.parent.joinpath('encoder_model.onnx'), onnx_outputs, args.atol if args.atol else encoder_onnx_config.atol_for_validation)
        preprocessor = AutoTokenizer.from_pretrained(args.model)
        onnx_inputs, onnx_outputs = export(preprocessor, decoder_model, decoder_onnx_config, args.opset, args.output.parent.joinpath('decoder_model.onnx'))
        validate_model_outputs(decoder_onnx_config, preprocessor, decoder_model, args.output.parent.joinpath('decoder_model.onnx'), onnx_outputs, args.atol if args.atol else decoder_onnx_config.atol_for_validation)
        logger.info(f'All good, model saved at: {args.output.parent.joinpath('encoder_model.onnx').as_posix()}, {args.output.parent.joinpath('decoder_model.onnx').as_posix()}')
    else:
        if args.preprocessor == 'auto':
            preprocessor = get_preprocessor(args.model)
        elif args.preprocessor == 'tokenizer':
            preprocessor = AutoTokenizer.from_pretrained(args.model)
        elif args.preprocessor == 'image_processor':
            preprocessor = AutoImageProcessor.from_pretrained(args.model)
        elif args.preprocessor == 'feature_extractor':
            preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
        elif args.preprocessor == 'processor':
            preprocessor = AutoProcessor.from_pretrained(args.model)
        else:
            raise ValueError(f"Unknown preprocessor type '{args.preprocessor}'")
        if args.opset is None:
            args.opset = onnx_config.default_onnx_opset
        if args.opset < onnx_config.default_onnx_opset:
            raise ValueError(f'Opset {args.opset} is not sufficient to export {model_kind}. At least  {onnx_config.default_onnx_opset} is required.')
        onnx_inputs, onnx_outputs = export(preprocessor, model, onnx_config, args.opset, args.output)
        if args.atol is None:
            args.atol = onnx_config.atol_for_validation
        validate_model_outputs(onnx_config, preprocessor, model, args.output, onnx_outputs, args.atol)
        logger.info(f'All good, model saved at: {args.output.as_posix()}')
        warnings.warn('The export was done by transformers.onnx which is deprecated and will be removed in v5. We recommend using optimum.exporters.onnx in future. You can find more information here: https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model.', FutureWarning)