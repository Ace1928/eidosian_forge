from tensorflow.keras import layers
def get_conv(shape):
    return [layers.Conv1D, layers.Conv2D, layers.Conv3D][len(shape) - 3]