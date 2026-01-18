from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
def setImageInput(baseOperation, arrayType=None, dimNames=DIMENSION_NAMES, pixelName='pixels', typeName=None):
    """Determine how to convert "pixels" into an image-compatible argument"""
    baseOperation = asWrapper(baseOperation)
    rank = len([argName for argName in baseOperation.argNames if argName in dimNames]) + 1
    if arrayType:
        converter = TypedImageInputConverter(rank, pixelName, arrayType, typeName=typeName)
        for i, argName in enumerate(baseOperation.argNames):
            if argName in dimNames:
                baseOperation.setPyConverter(argName)
                baseOperation.setCConverter(argName, getattr(converter, argName))
            elif argName == 'type' and typeName is not None:
                baseOperation.setPyConverter(argName)
                baseOperation.setCConverter(argName, converter.type)
    else:
        converter = ImageInputConverter(rank, pixelsName=pixelName, typeName=typeName or 'type')
    for argName in baseOperation.argNames:
        if argName in DATA_SIZE_NAMES:
            baseOperation.setPyConverter(argName)
            baseOperation.setCConverter(argName, converter.imageDataSize)
    baseOperation.setPyConverter(pixelName, converter)
    return baseOperation