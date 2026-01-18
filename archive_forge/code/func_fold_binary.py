import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def fold_binary(self, name, value):
    """+
        Headers are folded using the Header folding algorithm, which preserves
        existing line breaks in the value, and wraps each resulting line to the
        max_line_length.  If cte_type is 7bit, non-ascii binary data is CTE
        encoded using the unknown-8bit charset.  Otherwise the original source
        header is used, with its existing line breaks and/or binary data.

        """
    folded = self._fold(name, value, sanitize=self.cte_type == '7bit')
    return folded.encode('ascii', 'surrogateescape')