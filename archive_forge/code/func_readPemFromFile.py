import base64
import sys
def readPemFromFile(fileObj, startMarker='-----BEGIN CERTIFICATE-----', endMarker='-----END CERTIFICATE-----'):
    idx, substrate = readPemBlocksFromFile(fileObj, (startMarker, endMarker))
    return substrate