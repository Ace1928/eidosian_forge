import collections
def is_train(mode):
    return mode in [KerasModeKeys.TRAIN, EstimatorModeKeys.TRAIN]