from collections import namedtuple
def table_names(self, cursor=None, include_views=False):
    """
        Return a list of names of all tables that exist in the database.
        Sort the returned table list by Python's default sorting. Do NOT use
        the database's ORDER BY here to avoid subtle differences in sorting
        order between databases.
        """

    def get_names(cursor):
        return sorted((ti.name for ti in self.get_table_list(cursor) if include_views or ti.type == 't'))
    if cursor is None:
        with self.connection.cursor() as cursor:
            return get_names(cursor)
    return get_names(cursor)