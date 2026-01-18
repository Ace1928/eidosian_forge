import os, platform
def expand_relator(word):
    syllables = []
    for letter in word:
        if letter.isupper():
            syllables.append('%s^-1' % letter.lower())
        else:
            syllables.append(letter)
    return '*'.join(syllables)