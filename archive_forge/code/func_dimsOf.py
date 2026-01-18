import ctypes, _ctypes
from OpenGL.raw.GL import _types 
from OpenGL.arrays import _arrayconstants as GL_1_1
from OpenGL import constant, error
from OpenGL._configflags import ERROR_ON_COPY
from OpenGL.arrays import formathandler
from OpenGL._bytes import bytes,unicode,as_8_bit
import operator
@classmethod
def dimsOf(cls, x):
    """Calculate total dimension-set of the elements in x
        
        This is *extremely* messy, as it has to track nested arrays
        where the arrays could be different sizes on all sorts of 
        levels...
        """
    try:
        dimensions = [len(x)]
    except (TypeError, AttributeError, ValueError) as err:
        return []
    else:
        childDimension = None
        for child in x:
            newDimension = cls.dimsOf(child)
            if childDimension is not None:
                if newDimension != childDimension:
                    raise ValueError('Non-uniform array encountered: %s versus %s' % (newDimension, childDimension), x)