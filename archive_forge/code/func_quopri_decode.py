import codecs
import quopri
from io import BytesIO
def quopri_decode(input, errors='strict'):
    assert errors == 'strict'
    f = BytesIO(input)
    g = BytesIO()
    quopri.decode(f, g)
    return (g.getvalue(), len(input))