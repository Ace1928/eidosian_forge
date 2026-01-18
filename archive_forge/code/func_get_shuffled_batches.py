import random
import pytest
from thinc.api import (
def get_shuffled_batches(Xs, Ys, batch_size):
    zipped = list(zip(Xs, Ys))
    random.shuffle(zipped)
    for i in range(0, len(zipped), batch_size):
        batch_X, batch_Y = zip(*zipped[i:i + batch_size])
        yield (list(batch_X), list(batch_Y))