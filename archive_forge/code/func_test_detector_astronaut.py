import skimage.data as data
from skimage.feature import Cascade
def test_detector_astronaut():
    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(trained_file)
    img = data.astronaut()
    detected = detector.detect_multi_scale(img=img, scale_factor=1.2, step_ratio=1, min_size=(60, 60), max_size=(123, 123))
    assert len(detected) == 1, 'One face should be detected.'