import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from huggingface_hub import model_info
from ..configuration_utils import PretrainedConfig
from ..dynamic_module_utils import get_class_from_dynamic_module
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from ..models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
from ..models.auto.modeling_auto import AutoModelForDepthEstimation, AutoModelForImageToImage
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
from .audio_classification import AudioClassificationPipeline
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .base import (
from .conversational import Conversation, ConversationalPipeline
from .depth_estimation import DepthEstimationPipeline
from .document_question_answering import DocumentQuestionAnsweringPipeline
from .feature_extraction import FeatureExtractionPipeline
from .fill_mask import FillMaskPipeline
from .image_classification import ImageClassificationPipeline
from .image_feature_extraction import ImageFeatureExtractionPipeline
from .image_segmentation import ImageSegmentationPipeline
from .image_to_image import ImageToImagePipeline
from .image_to_text import ImageToTextPipeline
from .mask_generation import MaskGenerationPipeline
from .object_detection import ObjectDetectionPipeline
from .question_answering import QuestionAnsweringArgumentHandler, QuestionAnsweringPipeline
from .table_question_answering import TableQuestionAnsweringArgumentHandler, TableQuestionAnsweringPipeline
from .text2text_generation import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .text_to_audio import TextToAudioPipeline
from .token_classification import (
from .video_classification import VideoClassificationPipeline
from .visual_question_answering import VisualQuestionAnsweringPipeline
from .zero_shot_audio_classification import ZeroShotAudioClassificationPipeline
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline
from .zero_shot_object_detection import ZeroShotObjectDetectionPipeline
def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"depth-estimation"`
            - `"document-question-answering"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"image-feature-extraction"`
            - `"image-segmentation"`
            - `"image-to-text"`
            - `"image-to-image"`
            - `"object-detection"`
            - `"question-answering"`
            - `"summarization"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"text-generation"`
            - `"text-to-audio"` (alias `"text-to-speech"` available)
            - `"token-classification"` (alias `"ner"` available)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"video-classification"`
            - `"visual-question-answering"` (alias `"vqa"` available)
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
            - `"zero-shot-object-detection"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "translation_XX_to_YY"


    """
    return PIPELINE_REGISTRY.check_task(task)