from tensorflow.keras import layers
def get_sep_conv(shape):
    return [layers.SeparableConv1D, layers.SeparableConv2D, layers.Conv3D][len(shape) - 3]