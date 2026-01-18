from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def stripspace(string: str) -> str:
    result = ''
    last_was_space = True
    for character in string:
        if character.isspace():
            if not last_was_space:
                result += ' '
            last_was_space = True
        else:
            result += character
            last_was_space = False
    return result.strip()