from skimage.feature import multiscale_basic_features
def compute_features(self, image):
    if self.features_func is None:
        self.features_func = multiscale_basic_features
    self.features = self.features_func(image)