def get_max_width(self):
    """Get the maximum image frame width.

        This method is useful for determining texture space requirements: due
        to the use of ``anchor_x`` the actual required playback area may be
        larger.

        :rtype: int
        """
    return max([frame.image.width for frame in self.frames])