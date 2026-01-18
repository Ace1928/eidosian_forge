import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
def skip_headers(self, input_file):
    """Skip headers in `input_file`.

        Parameters
        ----------
        input_file : file
            File object.

        """
    for lineno, _ in enumerate(input_file):
        if lineno == 2:
            break