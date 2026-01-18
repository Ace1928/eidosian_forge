def get_max_height(self):
    """Get the maximum image frame height.

        This method is useful for determining texture space requirements: due
        to the use of ``anchor_y`` the actual required playback area may be
        larger.

        :rtype: int
        """
    return max([frame.image.height for frame in self.frames])