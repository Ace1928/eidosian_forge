from unittest import mock
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
def test_img_clf_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.ImageClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.generate_data(num_instances=100, shape=(32, 32, 3)), y=test_utils.generate_one_hot_labels(num_instances=100, num_classes=10))
    assert fit.is_called