from keras.src import backend
from keras.src import ops
def get_recurrent_dropout_mask(self, step_input):
    if not hasattr(self, '_recurrent_dropout_mask'):
        self._recurrent_dropout_mask = None
    if self._recurrent_dropout_mask is None and self.recurrent_dropout > 0:
        ones = ops.ones_like(step_input)
        self._recurrent_dropout_mask = backend.random.dropout(ones, rate=self.dropout, seed=self.seed_generator)
    return self._recurrent_dropout_mask