import math
import warnings
import matplotlib.dates
def convert_symbol(mpl_symbol):
    """Convert mpl marker symbol to plotly symbol and return symbol."""
    if isinstance(mpl_symbol, list):
        symbol = list()
        for s in mpl_symbol:
            symbol += [convert_symbol(s)]
        return symbol
    elif mpl_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[mpl_symbol]
    else:
        return 'circle'