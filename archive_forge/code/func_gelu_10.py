import math
import tensorflow as tf
from packaging.version import parse
def gelu_10(x):
    """
    Clip the range of possible GeLU outputs between [-10, 10]. This is especially useful for quantization purpose, as
    it allows mapping 2 negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see
    https://arxiv.org/abs/1606.08415 :param x: :return:
    """
    return tf.clip_by_value(_gelu(x), -10, 10)