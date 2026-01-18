from tensorflow.keras import layers
def get_global_max_pooling(shape):
    return [layers.GlobalMaxPool1D, layers.GlobalMaxPool2D, layers.GlobalMaxPool3D][len(shape) - 3]