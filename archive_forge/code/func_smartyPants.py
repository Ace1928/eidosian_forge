from a single quote by the algorithm. Therefore, a text like::
import re, sys
def smartyPants(text, attr=default_smartypants_attr, language='en'):
    """Main function for "traditional" use."""
    return ''.join([t for t in educate_tokens(tokenize(text), attr, language)])