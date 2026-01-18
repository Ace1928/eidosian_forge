from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
def forward_multimodal(self, text_features: ndarray, attention_mask: ndarray, image_features: ndarray) -> Tuple[ndarray, ndarray]:
    return self.reranker_session.run(None, {'text_features': text_features, 'attention_mask': attention_mask, 'image_features': image_features})