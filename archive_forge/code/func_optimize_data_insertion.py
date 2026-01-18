import sqlite3
def optimize_data_insertion(conn, data):
    """
    Optimizes data insertion into a database using efficient methods.

    Args:
    conn (sqlite3.Connection): The database connection object.
    data (list of tuple): The data to be inserted into the database.
    """
    if conn is None or not data:
        return 'No connection or data to insert.'
    try:
        cursor = conn.cursor()
        query = 'INSERT INTO your_table (column1, column2) VALUES (?, ?)'
        cursor.executemany(query, data)
        conn.commit()
        return 'Data inserted successfully.'
    except sqlite3.Error as e:
        return f'Database error: {e}'