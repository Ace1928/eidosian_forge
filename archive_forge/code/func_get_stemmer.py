import warnings
import snowballstemmer
from sphinx.deprecation import RemovedInSphinx70Warning
def get_stemmer() -> BaseStemmer:
    warnings.warn("get_stemmer() is deprecated, use snowballstemmer.stemmer('porter') instead.", RemovedInSphinx70Warning, stacklevel=2)
    return PyStemmer()