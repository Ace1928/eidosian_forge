from collections import namedtuple
import cv2
def testOverwriteNativeMethod(arg):
    return NativeMethodPatchedResult(arg + 1, cv2.utils._native.testOverwriteNativeMethod(arg))