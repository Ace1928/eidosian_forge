import typing as t
def simpleString(self) -> str:
    return f'struct<{', '.join((x.simpleString() for x in self))}>'