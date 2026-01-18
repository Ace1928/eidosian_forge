import pickle
import re
from debian.deprecation import function_deprecated_by
def correlations(self):
    """
        Generate the list of correlation as a tuple (hastag, hasalsotag, score).

        Every tuple will indicate that the tag 'hastag' tends to also
        have 'hasalsotag' with a score of 'score'.
        """
    for pivot in self.iter_tags():
        with_ = self.filter_packages_tags(lambda pt: pivot in pt[1])
        without = self.filter_packages_tags(lambda pt: pivot not in pt[1])
        for tag in with_.iter_tags():
            if tag == pivot:
                continue
            has = float(with_.card(tag)) / float(with_.package_count())
            hasnt = float(without.card(tag)) / float(without.package_count())
            yield (pivot, tag, has - hasnt)