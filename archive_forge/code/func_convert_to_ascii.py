import re
def convert_to_ascii(self, text: str) -> str:
    """
        Converts unicode to ascii
        """
    return text.encode('ascii', 'ignore').decode('utf-8')