from OpenGL import contextdata
from OpenGL.GL.VERSION import GL_1_1 as _simple
def parseFeedback(buffer, entryCount):
    """Parse the feedback buffer into Python object records"""
    bufferIndex = 0
    result = []
    getVertex = createGetVertex()
    while bufferIndex < entryCount:
        token = int(buffer[bufferIndex])
        bufferIndex += 1
        if token in SINGLE_VERTEX_TOKENS:
            vData, bufferIndex = getVertex(buffer, bufferIndex)
            result.append((SINGLE_VERTEX_TOKENS.get(token), Vertex(*vData)))
        elif token in DOUBLE_VERTEX_TOKENS:
            vData, bufferIndex = getVertex(buffer, bufferIndex)
            vData2, bufferIndex = getVertex(buffer, bufferIndex)
            result.append((DOUBLE_VERTEX_TOKENS.get(token), Vertex(*vData), Vertex(*vData2)))
        elif token == _simple.GL_PASS_THROUGH_TOKEN:
            result.append((_simple.GL_PASS_THROUGH_TOKEN, buffer[bufferIndex]))
            bufferIndex += 1
        elif token == _simple.GL_POLYGON_TOKEN:
            temp = [_simple.GL_POLYGON_TOKEN]
            count = int(buffer[bufferIndex])
            bufferIndex += 1
            for item in range(count):
                vData, bufferIndex = getVertex(buffer, bufferIndex)
                temp.append(Vertex(*vData))
            result.append(tuple(temp))
        else:
            raise ValueError('Unrecognised token %r in feedback stream' % (token,))
    return result