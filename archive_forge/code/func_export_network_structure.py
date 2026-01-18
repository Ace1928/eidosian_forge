import csv
from config import global_config
from logger import CustomLogger
def export_network_structure(self, network_structure, filename='network_structure.csv'):
    """
        Exports the structure of the neural network to a CSV file.

        Parameters:
            network_structure (dict): The structure of the neural network to export.
            filename (str): The name of the file to export the structure to.
        """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=self.config['delimiter'], quotechar=self.config['quotechar'], quoting=csv.QUOTE_MINIMAL)
            for layer, info in network_structure.items():
                writer.writerow([layer, info['Hexagons'], info['Activation_Function']])
        self.logger.info(f'Network structure successfully exported to {filename}.')
    except Exception as e:
        self.logger.error(f'Failed to export network structure to {filename}. Error: {e}')