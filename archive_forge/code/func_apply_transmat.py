import warnings
from collections import OrderedDict
import numpy as np
from gensim import utils
def apply_transmat(self, words_space):
    """Map the source word vector to the target word vector using translation matrix.

        Parameters
        ----------
        words_space : :class:`~gensim.models.translation_matrix.Space`
            `Space` object constructed for the words to be translated.

        Returns
        -------
        :class:`~gensim.models.translation_matrix.Space`
            `Space` object constructed for the mapped words.

        """
    return Space(np.dot(words_space.mat, self.translation_matrix), words_space.index2word)