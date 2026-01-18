import keras.src as keras
from keras.src.testing_infra import test_utils
def get_multi_io_subclass_model(use_bn=False, use_dp=False, num_classes=(2, 3)):
    """Creates MultiIOModel for the tests of subclass model."""
    shared_layer = keras.layers.Dense(32, activation='relu')
    branch_a = [shared_layer]
    if use_dp:
        branch_a.append(keras.layers.Dropout(0.5))
    branch_a.append(keras.layers.Dense(num_classes[0], activation='softmax'))
    branch_b = [shared_layer]
    if use_bn:
        branch_b.append(keras.layers.BatchNormalization())
    branch_b.append(keras.layers.Dense(num_classes[1], activation='softmax'))
    model = test_utils._MultiIOSubclassModel(branch_a, branch_b, name='test_model')
    return model