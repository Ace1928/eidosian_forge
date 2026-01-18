@classmethod
def from_image_sequence(cls, sequence, duration, loop=True):
    """Create an animation from a list of images and a constant framerate.

        :Parameters:
            `sequence` : list of `~pyglet.image.AbstractImage`
                Images that make up the animation, in sequence.
            `duration` : float
                Number of seconds to display each image.
            `loop` : bool
                If True, the animation will loop continuously.

        :rtype: :py:class:`~pyglet.image.Animation`
        """
    frames = [AnimationFrame(image, duration) for image in sequence]
    if not loop:
        frames[-1].duration = None
    return cls(frames)