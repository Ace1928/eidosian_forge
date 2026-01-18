import contextlib
import tempfile
import os
import shutil
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
def get_tmpfile(suffix):
    """Get full path to file `suffix` in temporary folder.
    This function doesn't creates file (only generate unique name).
    Also, it may return different paths in consecutive calling.

    Parameters
    ----------
    suffix : str
        Suffix of file.

    Returns
    -------
    str
        Path to `suffix` file in temporary folder.

    Examples
    --------
    Using this function we may get path to temporary file and use it, for example, to store temporary model.

    .. sourcecode:: pycon

        >>> from gensim.models import LsiModel
        >>> from gensim.test.utils import get_tmpfile, common_dictionary, common_corpus
        >>>
        >>> tmp_f = get_tmpfile("toy_lsi_model")
        >>>
        >>> model = LsiModel(common_corpus, id2word=common_dictionary)
        >>> model.save(tmp_f)
        >>>
        >>> loaded_model = LsiModel.load(tmp_f)

    """
    return os.path.join(tempfile.mkdtemp(), suffix)