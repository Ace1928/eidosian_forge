from re import finditer
from xml.sax.saxutils import escape, unescape
def align_tokens(tokens, sentence):
    """
    This module attempt to find the offsets of the tokens in *s*, as a sequence
    of ``(start, end)`` tuples, given the tokens and also the source string.

        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> from nltk.tokenize.util import align_tokens
        >>> s = str("The plane, bound for St Petersburg, crashed in Egypt's "
        ... "Sinai desert just 23 minutes after take-off from Sharm el-Sheikh "
        ... "on Saturday.")
        >>> tokens = TreebankWordTokenizer().tokenize(s)
        >>> expected = [(0, 3), (4, 9), (9, 10), (11, 16), (17, 20), (21, 23),
        ... (24, 34), (34, 35), (36, 43), (44, 46), (47, 52), (52, 54),
        ... (55, 60), (61, 67), (68, 72), (73, 75), (76, 83), (84, 89),
        ... (90, 98), (99, 103), (104, 109), (110, 119), (120, 122),
        ... (123, 131), (131, 132)]
        >>> output = list(align_tokens(tokens, s))
        >>> len(tokens) == len(expected) == len(output)  # Check that length of tokens and tuples are the same.
        True
        >>> expected == list(align_tokens(tokens, s))  # Check that the output is as expected.
        True
        >>> tokens == [s[start:end] for start, end in output]  # Check that the slices of the string corresponds to the tokens.
        True

    :param tokens: The list of strings that are the result of tokenization
    :type tokens: list(str)
    :param sentence: The original string
    :type sentence: str
    :rtype: list(tuple(int,int))
    """
    point = 0
    offsets = []
    for token in tokens:
        try:
            start = sentence.index(token, point)
        except ValueError as e:
            raise ValueError(f'substring "{token}" not found in "{sentence}"') from e
        point = start + len(token)
        offsets.append((start, point))
    return offsets