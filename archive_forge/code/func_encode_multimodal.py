from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
def encode_multimodal(self, image: Optional[ndarray]=None, text: Dict[str, ndarray]=None, image_features: Optional[ndarray]=None, text_features: Optional[ndarray]=None, attention_mask: Optional[ndarray]=None, return_scores: bool=False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    """Passes preprocessed texts (or precomputed texts features) and
            preprocessed images (or precomputed images features) through multimodal encoded to produce matching scores and optionally multimodal joint embeddings.

        :param image: Preprocessed images
        :param text: Preprocessed texts
        :param image_features: Precomputed images features
        :param text_features: Precomputed text features
        :param attention_mask: Attention masks, not required if pass `text` instead of text_features
        """
    assert image is not None or image_features is not None, 'Either `image` or `image_features` should be non None'
    assert text is not None or text_features is not None, 'Either `text_data` or `text_features` should be non None'
    if text_features is not None:
        assert attention_mask is not None, 'if `text_features` is not None, then you should pass `attention_mask`'
    if image_features is None:
        image_features = self.image_encoder(image)
    if text_features is None:
        text_features = self.text_encoder(text['input_ids'], text['attention_mask'])
    matching_scores, embeddings = self.text_encoder.forward_multimodal(text_features, attention_mask if attention_mask is not None else text['attention_mask'], image_features)
    if return_scores:
        return (matching_scores, embeddings)
    return embeddings