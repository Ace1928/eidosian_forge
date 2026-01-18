from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@db.setter
def db(self, val):
    self['db'] = val