import sys
import select
def disable_glut(self):
    """Disable event loop integration with glut.

        This sets PyOS_InputHook to NULL and set the display function to a
        dummy one and set the timer to a dummy timer that will be triggered
        very far in the future.
        """
    import OpenGL.GLUT as glut
    from glut_support import glutMainLoopEvent
    glut.glutHideWindow()
    glutMainLoopEvent()
    self.clear_inputhook()