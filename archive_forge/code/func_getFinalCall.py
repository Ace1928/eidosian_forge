from OpenGL import acceleratesupport
def getFinalCall(self):
    """Retrieve and/or bind and retrieve final call"""
    if not self._finalCall:
        self._finalCall = self.finalise()
    return self._finalCall