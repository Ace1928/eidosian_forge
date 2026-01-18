import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_sorted(self, names, reverse=False):
    """
        Sort a list of CPU features ordered by the lowest interest.

        Parameters
        ----------
        'names': sequence
            sequence of supported feature names in uppercase.
        'reverse': bool, optional
            If true, the sorted features is reversed. (highest interest)

        Returns
        -------
        list, sorted CPU features
        """

    def sort_cb(k):
        if isinstance(k, str):
            return self.feature_supported[k]['interest']
        rank = max([self.feature_supported[f]['interest'] for f in k])
        rank += len(k) - 1
        return rank
    return sorted(names, reverse=reverse, key=sort_cb)