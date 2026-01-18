import os
def cinput(text, color=Colors.RESET, **kwargs):
    return input(color + text + Colors.RESET, **kwargs)