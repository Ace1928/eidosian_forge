import re
def collapse_whitespace(self, text: str) -> str:
    """
        Removes multiple whitespaces
        """
    return re.sub(re.compile('\\s+'), ' ', text)