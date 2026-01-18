from keras.src.applications import imagenet_utils
from keras.src.applications import resnet
from tensorflow.python.util.tf_export import keras_export
def stack_fn(x):
    x = resnet.stack2(x, 64, 3, name='conv2')
    x = resnet.stack2(x, 128, 8, name='conv3')
    x = resnet.stack2(x, 256, 36, name='conv4')
    return resnet.stack2(x, 512, 3, stride1=1, name='conv5')