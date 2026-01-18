import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import sparse
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def get_previous_ngrams(self, input_ids: jnp.ndarray, vocab_size: int, cur_len: int):
    """
        get a matrix of size (batch_size,) + (vocab_size,)*n (for n-grams) that
        represent the n-grams that occured previously.
        The BCOO representation allow to store only the few non-zero entries, instead of the full (huge) matrix
        """
    batch_size, seq_len = input_ids.shape
    seq_ngrams = seq_len - (self.ngram_size - 1)
    cur_ngrams = cur_len - (self.ngram_size - 1)

    def body_fun(i, val):
        b = i % batch_size
        pos = i // batch_size
        return val.at[i].set(jnp.array([b] + [jnp.array(input_ids)[b, pos + j] for j in range(self.ngram_size)]))
    shape = (batch_size * seq_ngrams, self.ngram_size + 1)
    all_update_indices = jax.lax.fori_loop(0, batch_size * cur_ngrams, body_fun, jnp.zeros(shape, dtype=input_ids.dtype))
    data = (jnp.arange(batch_size * seq_ngrams) < batch_size * cur_ngrams).astype('float32')
    return sparse.BCOO((data, all_update_indices), shape=(batch_size,) + (vocab_size,) * self.ngram_size)