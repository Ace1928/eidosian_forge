from typing import Any, Dict, Optional, Union
from transformers import (
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import SUPPORTED_TASKS as TRANSFORMERS_SUPPORTED_TASKS
from transformers.pipelines import infer_framework_load_model
from ..bettertransformer import BetterTransformer
from ..utils import is_onnxruntime_available
from ..utils.file_utils import find_files_matching_pattern
def load_ort_pipeline(model, targeted_task, load_tokenizer, tokenizer, feature_extractor, load_feature_extractor, SUPPORTED_TASKS, subfolder: str='', token: Optional[Union[bool, str]]=None, revision: str='main', model_kwargs: Optional[Dict[str, Any]]=None, config: AutoConfig=None, **kwargs):
    if model_kwargs is None:
        model_kwargs = {}
    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]['default']
        model = SUPPORTED_TASKS[targeted_task]['class'][0].from_pretrained(model_id, export=True)
    elif isinstance(model, str):
        from ..onnxruntime.modeling_seq2seq import ENCODER_ONNX_FILE_PATTERN, ORTModelForConditionalGeneration
        model_id = model
        ort_model_class = SUPPORTED_TASKS[targeted_task]['class'][0]
        if issubclass(ort_model_class, ORTModelForConditionalGeneration):
            pattern = ENCODER_ONNX_FILE_PATTERN
        else:
            pattern = '.+?.onnx'
        onnx_files = find_files_matching_pattern(model, pattern, glob_pattern='**/*.onnx', subfolder=subfolder, use_auth_token=token, revision=revision)
        export = len(onnx_files) == 0
        model = ort_model_class.from_pretrained(model, export=export, **model_kwargs)
    elif isinstance(model, ORTModel):
        if tokenizer is None and load_tokenizer:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                    tokenizer = preprocessor
                    break
            if tokenizer is None:
                raise ValueError('Could not automatically find a tokenizer for the ORTModel, you must pass a tokenizer explictly')
        if feature_extractor is None and load_feature_extractor:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, SequenceFeatureExtractor):
                    feature_extractor = preprocessor
                    break
            if feature_extractor is None:
                raise ValueError('Could not automatically find a feature extractor for the ORTModel, you must pass a feature_extractor explictly')
        model_id = None
    else:
        raise ValueError(f'Model {model} is not supported. Please provide a valid model either as string or ORTModel.\n            You can also provide non model then a default one will be used')
    return (model, model_id, tokenizer, feature_extractor)