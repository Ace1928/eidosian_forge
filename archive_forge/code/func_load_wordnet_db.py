import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def load_wordnet_db(ensure_compatibility):
    """
        Load WordNet database to apply specific rules instead of stemming + load file for special cases to ensure kind of compatibility (at list with DUC 2004) with the original stemmer used in the Perl script
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)

        Raises:
            FileNotFoundError: If one of both databases is not found
        """
    files_to_load = [Rouge.WORDNET_DB_FILEPATH]
    if ensure_compatibility:
        files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)
    for wordnet_db in files_to_load:
        filepath = pkg_resources.resource_filename(__name__, wordnet_db)
        if not os.path.exists(filepath):
            raise FileNotFoundError("The file '{}' does not exist".format(filepath))
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp:
                k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
                assert k not in Rouge.WORDNET_KEY_VALUE
                Rouge.WORDNET_KEY_VALUE[k] = v