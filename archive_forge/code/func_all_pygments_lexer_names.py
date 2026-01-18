from typing import List
import pytest
import pygments.lexers
import pygments.lexer
from IPython.lib.lexers import IPythonConsoleLexer, IPythonLexer, IPython3Lexer
@pytest.fixture
def all_pygments_lexer_names() -> List[str]:
    """Get all lexer names registered in pygments."""
    return {l[0] for l in pygments.lexers.get_all_lexers()}