import sqlite3
from tkinter import messagebox, Tk, Label, Entry, Button, StringVar
def create_gui():
    """
    Creates a graphical user interface for the application.
    """
    window = Tk()
    window.title('Extract Messages')
    Label(window, text='Contact Number').grid(row=0, column=0)
    contact_number_var = StringVar()
    contact_entry = Entry(window, textvariable=contact_number_var)
    contact_entry.grid(row=0, column=1)
    Label(window, text='Output Format').grid(row=1, column=0)
    output_format_var = StringVar()
    output_format_entry = Entry(window, textvariable=output_format_var)
    output_format_entry.grid(row=1, column=1)
    Button(window, text='Extract', command=lambda: extract_messages(None, contact_number_var.get(), output_format_var.get())).grid(row=2, columnspan=2)
    window.mainloop()