import enum
import logging
import os
import types
import typing
@property
def extern_def(self) -> str:
    return '{} {}({})'.format(self.rtype, self.name, ' ,'.join(self.arguments))