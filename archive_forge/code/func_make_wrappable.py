import re
import html
from paste.util import PySourceColor
def make_wrappable(html, wrap_limit=60, split_on=';?&@!$#-/\\"\''):
    if len(html) <= wrap_limit:
        return html
    words = html.split()
    new_words = []
    for word in words:
        wrapped_word = ''
        while len(word) > wrap_limit:
            for char in split_on:
                if char in word:
                    first, rest = word.split(char, 1)
                    wrapped_word += first + char + '<wbr>'
                    word = rest
                    break
            else:
                for i in range(0, len(word), wrap_limit):
                    wrapped_word += word[i:i + wrap_limit] + '<wbr>'
                word = ''
        wrapped_word += word
        new_words.append(wrapped_word)
    return ' '.join(new_words)