import re
import unicodedata
from typing import Iterator
from emoji import unicode_codes
from emoji.tokenizer import Token, EmojiMatch, EmojiMatchZWJ, EmojiMatchZWJNonRGI, tokenize, filter_tokens
def distinct_emoji_list(string):
    """Returns distinct list of emojis from the string."""
    distinct_list = list({e['emoji'] for e in emoji_list(string)})
    return distinct_list