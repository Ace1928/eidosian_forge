from unittest import mock
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
def test_img_obj_det_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.tasks.image.ImageObjectDetector(directory=tmp_path, seed=test_utils.SEED)
    images, labels = test_utils.get_object_detection_data()
    auto_model.fit(x=images, y=labels)
    assert fit.is_called