from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
@staticmethod
def symbolsheet(size=20):
    """
        Open a new Tkinter window that displays the entire alphabet
        for the symbol font.  This is useful for constructing the
        ``SymbolWidget.SYMBOLS`` dictionary.
        """
    top = Tk()

    def destroy(e, top=top):
        top.destroy()
    top.bind('q', destroy)
    Button(top, text='Quit', command=top.destroy).pack(side='bottom')
    text = Text(top, font=('helvetica', -size), width=20, height=30)
    text.pack(side='left')
    sb = Scrollbar(top, command=text.yview)
    text['yscrollcommand'] = sb.set
    sb.pack(side='right', fill='y')
    text.tag_config('symbol', font=('symbol', -size))
    for i in range(256):
        if i in (0, 10):
            continue
        for k, v in list(SymbolWidget.SYMBOLS.items()):
            if v == chr(i):
                text.insert('end', '%-10s\t' % k)
                break
        else:
            text.insert('end', '%-10d  \t' % i)
        text.insert('end', '[%s]\n' % chr(i), 'symbol')
    top.mainloop()