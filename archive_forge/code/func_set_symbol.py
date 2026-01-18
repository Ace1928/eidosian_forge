from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def set_symbol(self, symbol):
    """
        Change the symbol that is displayed by this symbol widget.

        :type symbol: str
        :param symbol: The name of the symbol to display.
        """
    if symbol not in SymbolWidget.SYMBOLS:
        raise ValueError('Unknown symbol: %s' % symbol)
    self._symbol = symbol
    self.set_text(SymbolWidget.SYMBOLS[symbol])